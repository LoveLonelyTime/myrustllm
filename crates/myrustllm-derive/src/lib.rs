use proc_macro::TokenStream;
use proc_macro_crate::{FoundCrate, crate_name};
use quote::quote;
use syn::{DeriveInput, parse_macro_input};

#[proc_macro_derive(Module, attributes(module))]
pub fn derive_module(input: TokenStream) -> TokenStream {
    let found = crate_name("myrustllm").unwrap();
    let crate_ident = match found {
        FoundCrate::Itself => quote!(crate),
        FoundCrate::Name(name) => {
            let ident = syn::Ident::new(&name, proc_macro2::Span::call_site());
            quote!(::#ident)
        }
    };

    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident;
    let generics = input.generics;
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
    let fields = match input.data {
        syn::Data::Struct(data) => data.fields,
        _ => panic!("Module can only be derived for structs."),
    };

    fn has_module_attr(field: &syn::Field) -> bool {
        field
            .attrs
            .iter()
            .any(|attr| attr.path().is_ident("module"))
    }

    let visits = fields.iter().filter(|f| has_module_attr(f)).map(|f| {
        let ident = &f.ident;
        quote! {
            self.#ident.visit(v);
        }
    });

    let expanded = quote! {
        impl #impl_generics #crate_ident::nn::module::Module for #name #ty_generics #where_clause {
            fn visit<'a>(&'a mut self, v: &mut dyn #crate_ident::nn::module::ModuleVisitor<'a>) {
                #( #visits )*
            }
        }
    };

    TokenStream::from(expanded)
}
